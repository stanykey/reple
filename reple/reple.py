#!/usr/bin/env python3
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from importlib import import_module
from json import load
from operator import add
from os import makedirs
from os import path
from os import system
from sys import argv
from typing import Any

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import style_from_pygments_cls
from pygments.lexer import Lexer
from pygments.styles import get_style_by_name


class CompilationEnvironment:
    def __init__(self, compile_command_template: str, compile_args: dict[str, str], user_cargs: dict[str, str]) -> None:
        self._compile = compile_command_template
        self.compile_args = defaultdict(str, compile_args)
        self.user_cargs = user_cargs

        assert isinstance(self._compile, str)
        assert isinstance(self.compile_args, dict)
        assert "{bin_fname}" in self._compile
        assert "{code_fname}" in self._compile

        self.code_suffix = self.compile_args["code_suffix"]
        self.bin_suffix = self.compile_args["bin_suffix"]

    # Construct a compile line
    def get_compile_line(self, code_fname: str, bin_fname: str) -> str:
        if "{user_cargs}" in self._compile:
            return self._compile.format(
                code_fname=code_fname, bin_fname=bin_fname, user_cargs=self.user_cargs, **self.compile_args
            )
        else:
            return self._compile.format(code_fname=code_fname, bin_fname=bin_fname, **self.compile_args)

    # Given a code string and file name,
    # attempt to compile
    def compile(self, code_string: str, name: str, output_dir: str) -> str | None:
        code_fname = output_dir + name + self.code_suffix
        bin_fname = output_dir + name + self.bin_suffix
        compile_line = self.get_compile_line(code_fname, bin_fname)

        f = open(code_fname, "w")
        f.write(code_string)
        f.close()

        system(compile_line)

        if path.isfile(bin_fname):
            return bin_fname
        else:
            return None


class RuntimeEnvironment:
    def __init__(self, runcommand: str, user_rargs: dict[str, str]) -> None:
        self.runcommand = runcommand
        self.user_rargs = user_rargs

    def get_run_line(self, bin_fname: str, output_fname: str) -> str:
        if "{user_rargs}" in self.runcommand:
            runcommand = self.runcommand.format(bin_fname=bin_fname, user_rargs=self.user_rargs)
        else:
            runcommand = self.runcommand.format(bin_fname=bin_fname)
        return " ".join([runcommand, bin_fname, ">", output_fname, "2>&1"])

    def run(self, bin_fname: str, output_fname: str) -> None:
        run_line = self.get_run_line(bin_fname, output_fname)
        system(run_line)


class CodeTemplate:
    def __init__(self, template: str, template_args: dict[str, str]) -> None:
        self.template = template
        self.template_args = template_args

        self.line_epilogue = self.template_args["line_epilogue"]

        assert "prolog_lines" in self.template
        assert "repl_lines" in self.template

    def generate_code(self, prolog_lines: list[str], repl_lines: list[str]) -> str:
        prolog_text = "\n".join(prolog_lines)
        repl_text = ("\n" + self.line_epilogue + "\n").join(repl_lines)
        return self.template.format(prolog_lines=prolog_text, repl_lines=repl_text, **self.template_args)


class Reple:
    def __init__(
        self,
        comp_env: CompilationEnvironment,
        runtime_env: RuntimeEnvironment,
        code_templ: CodeTemplate,
        lexer: type[Lexer] | None = None,
        output_dir: str = "/tmp/repl/",
        output_name: str = "repl",
        enclosers: list[tuple[str, ...]] | None = None,
        prolog_char: str = "$",
    ) -> None:
        assert isinstance(comp_env, CompilationEnvironment)
        assert isinstance(runtime_env, RuntimeEnvironment)
        assert isinstance(code_templ, CodeTemplate)
        self.comp_env = comp_env
        self.runtime_env = runtime_env
        self.code_templ = code_templ

        self.lexer = None if lexer is None else PygmentsLexer(lexer)
        self.output_dir = output_dir

        self.prolog_lines: list[str] = []
        self.repl_lines: list[str] = []
        self.in_prolog = False

        self.executions: defaultdict[int, list[str]] = defaultdict(list)

        self.output_name = output_name
        self.output_fname_nonce = 0

        self.enclosers = enclosers if enclosers else [("{", "}")]
        self.prolog_char = prolog_char

        self.style = get_style_by_name("native")
        self.history = InMemoryHistory()

        if not path.exists(self.output_dir):
            makedirs(self.output_dir)
        system(" ".join(["rm -rf", self.output_dir + "*"]))

    @staticmethod
    def count_enclosers(line: str, start: str, stop: str) -> int:
        return line.count(start) - line.count(stop)

    def get_fname(self) -> str:
        return self.output_name + str(self.output_fname_nonce)

    def append_lines(self, prolog_line: str, repl_line: str) -> None:
        if prolog_line != "":
            self.prolog_lines.append(prolog_line)
        if repl_line != "":
            self.repl_lines.append(repl_line)

    def get_new_lines(self) -> list[str]:
        new_lines = len(self.executions[self.output_fname_nonce]) - len(self.executions[self.output_fname_nonce - 1])
        if new_lines > 0:
            return self.executions[self.output_fname_nonce][-new_lines:]
        else:
            return []

    def execute(self, repl_line: str, prolog_line: str = "") -> None:
        cur_prolog_lines = self.prolog_lines + [prolog_line]
        cur_repl_lines = self.repl_lines + [repl_line]

        code = self.code_templ.generate_code(cur_prolog_lines, cur_repl_lines)
        bin_fname = self.comp_env.compile(code, self.get_fname(), self.output_dir)

        if bin_fname:
            output_fname = bin_fname + ".out"
            self.append_lines(prolog_line, repl_line)
            self.runtime_env.run(bin_fname, output_fname)

            self.executions[self.output_fname_nonce] = open(output_fname).readlines()
            new_lines = self.get_new_lines()
            if len(new_lines) > 0:
                for line in [line.strip() for line in new_lines]:
                    print(line)
            self.output_fname_nonce += 1

    def process_line(
        self, line: str, repl_lines: list[str], prolog_lines: list[str], encloser_counts: list[int]
    ) -> bool:
        if line == "clear":
            self.prolog_lines.clear()
            self.repl_lines.clear()
            self.executions.clear()
        elif line == "quit":
            return False
        elif len(line) <= 0:
            pass
        elif line[0] == self.prolog_char:
            if self.in_prolog:
                prolog_line = "\n".join(prolog_lines)
                self.execute("", prolog_line)
                self.in_prolog = False
                prolog_lines.clear()
                self.process_line(line[1:], repl_lines, prolog_lines, encloser_counts)
            else:
                self.in_prolog = True
                self.process_line(line[1:], repl_lines, prolog_lines, encloser_counts)
        elif line[-1] == self.prolog_char:
            if self.in_prolog:
                prolog_lines.append(line[:-1])
                prolog_line = "\n".join(prolog_lines)
                self.execute("", prolog_line)
                self.in_prolog = False
                prolog_lines.clear()
            else:
                self.process_line(line[:-1], repl_lines, prolog_lines, encloser_counts)
                self.process_line(line[-1:], repl_lines, prolog_lines, encloser_counts)
        elif self.in_prolog:
            prolog_lines.append(line)
        else:
            line_enclosers = [self.count_enclosers(line, x[0], x[1]) for x in self.enclosers]
            encloser_counts[:] = map(add, encloser_counts, line_enclosers)
            if sum(encloser_counts) <= 0:
                repl_lines.append(line)
                repl_line = "\n".join(repl_lines)
                self.execute(repl_line)
                repl_lines.clear()
            else:
                repl_lines.append(line)
        return True

    def run(self) -> None:
        repl_lines: list[str] = []
        prolog_lines: list[str] = []
        self.in_prolog = False
        encloser_counts = [0] * len(self.enclosers)
        while True:
            try:
                prompt_str = "... " if (self.in_prolog or sum(encloser_counts) > 0) else ">>> "
                line = prompt(
                    prompt_str,
                    lexer=self.lexer,
                    style=style_from_pygments_cls(get_style_by_name("native")),  # noqa
                    history=self.history,
                )
            except (KeyboardInterrupt, EOFError):
                break
            stat = self.process_line(line.rstrip(), repl_lines, prolog_lines, encloser_counts)
            if not stat:
                break


def configure_terminal_opts(terminal_opts: dict[str, Any]) -> dict[str, Any]:
    rterm_opts = {}
    if "lexer_class" in terminal_opts:
        lexer_class = import_module(terminal_opts["lexer_class"])
        lexer_fn = getattr(lexer_class, terminal_opts["lexer_fn"])
        rterm_opts["lexer"] = lexer_fn
    if "prolog_char" in terminal_opts:
        rterm_opts["prolog_char"] = terminal_opts["prolog_char"]
    if "enclosers" in terminal_opts:
        rterm_opts["enclosers"] = [tuple(x) for x in terminal_opts["enclosers"]]
    return rterm_opts


def get_config_fname(args: Namespace) -> str:
    reple_path = path.dirname(path.realpath(__file__))
    fname = reple_path + "/configs/"
    if args.fname is not None:
        fname += args.fname
    else:
        fname = reple_path + "/config/reple/" + args.env + ".json"
        if not path.isfile(fname):
            fname = reple_path + "/configs/" + args.env + ".json"
    return fname


def run_reple(cmd_args: list[str]) -> None:
    parser = ArgumentParser(
        description="reple, an interactive REPL \
            for executable-driven software toolchains."
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "-env",
        dest="env",
        type=str,
        help="reple\
            environment to use.  See $INSTALL_DIR/configs for included\
            environments.",
    )
    config_group.add_argument(
        "-f",
        dest="fname",
        type=str,
        help="File name for\
            the json config file",
    )
    parser.add_argument(
        "--rargs",
        dest="user_rargs",
        type=str,
        help="User\
            options to forward at runtime",
        default="",
    )
    parser.add_argument(
        "--cargs",
        dest="user_cargs",
        type=str,
        help="User\
            options to forward at compile time",
        default="",
    )
    args = parser.parse_args(cmd_args)

    fname = get_config_fname(args)
    config = load(open(fname))

    assert not (args.user_rargs != "" and "{user_rargs}" not in config["run"])
    assert not (args.user_cargs != "" and "{user_cargs}" not in config["compile"])

    comp_env = CompilationEnvironment(config["compile"], config["compile_args"], args.user_cargs)

    runtime_env = RuntimeEnvironment(config["run"], args.user_rargs)

    code_templ = CodeTemplate(config["template"], config["template_args"])

    terminal_opts = configure_terminal_opts(config["terminal_opts"])

    reple = Reple(comp_env, runtime_env, code_templ, **terminal_opts)

    reple.run()


def main() -> None:
    run_reple(argv[1:])


if __name__ == "__main__":
    main()
