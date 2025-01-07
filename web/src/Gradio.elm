module Gradio exposing (Config, Error(..), decodeOne, get)

import Http
import Json.Decode as D
import Json.Encode as E
import Parser exposing ((|.), (|=))
import Task
import Url.Builder


type alias Config =
    { host : String }


type Error
    = NetworkError String
    | ParsingError String
    | JsonError String
    | ApiError String


get : Config -> String -> List E.Value -> D.Decoder a -> (Result Error a -> msg) -> Cmd msg
get cfg path args decoder msg =
    start cfg path args
        |> Task.andThen (finish cfg path decoder)
        |> Task.attempt msg


start : Config -> String -> List E.Value -> Task.Task Error String
start cfg path args =
    Http.task
        { method = "POST"
        , headers = []
        , url =
            Url.Builder.crossOrigin cfg.host
                [ "gradio_api", "call", path ]
                []
        , body = Http.jsonBody (encodeArgs args)
        , resolver =
            Http.stringResolver
                (httpResolver
                    >> Result.andThen (jsonResolver eventIdDecoder)
                )
        , timeout = Nothing
        }


finish : Config -> String -> D.Decoder a -> String -> Task.Task Error a
finish cfg path decoder eventId =
    Http.task
        { method = "GET"
        , headers = []
        , url =
            Url.Builder.crossOrigin cfg.host
                [ "gradio_api", "call", path, eventId ]
                []
        , body = Http.emptyBody
        , resolver =
            Http.stringResolver
                (httpResolver
                    >> Result.andThen parsingResolver
                    >> Result.andThen (jsonResolver decoder)
                )
        , timeout = Nothing
        }


eventIdDecoder : D.Decoder String
eventIdDecoder =
    D.field "event_id" D.string


httpResolver : Http.Response String -> Result Error String
httpResolver response =
    case response of
        Http.GoodStatus_ _ body ->
            Ok body

        Http.BadUrl_ url ->
            Err (ApiError <| "Bad URL: " ++ url)

        Http.Timeout_ ->
            Err (NetworkError "Timed out")

        Http.NetworkError_ ->
            Err (NetworkError "Unknown network error")

        Http.BadStatus_ _ body ->
            Err (ApiError body)


parsingResolver : String -> Result Error String
parsingResolver raw =
    Parser.run eventParser raw
        |> Result.mapError (deadEndsToString >> ParsingError)


jsonResolver : D.Decoder a -> String -> Result Error a
jsonResolver decoder body =
    D.decodeString decoder body
        |> Result.mapError (D.errorToString >> JsonError)


encodeArgs : List E.Value -> E.Value
encodeArgs args =
    E.object [ ( "data", E.list identity args ) ]


decodeOne : D.Decoder a -> D.Decoder a
decodeOne decoder =
    D.list decoder
        |> D.map List.head
        |> D.andThen decodeOneHelper


decodeOneHelper : Maybe a -> D.Decoder a
decodeOneHelper maybe =
    case maybe of
        Just something ->
            D.succeed something

        Nothing ->
            D.fail "No result"



-- PARSER


eventParser : Parser.Parser String
eventParser =
    Parser.loop () eventParserHelper


eventParserHelper : () -> Parser.Parser (Parser.Step () String)
eventParserHelper _ =
    Parser.succeed identity
        |. Parser.keyword "event"
        |. Parser.symbol ":"
        |. Parser.spaces
        |= Parser.oneOf
            [ Parser.succeed (Parser.Loop ())
                |. Parser.keyword "heartbeat"
                |. Parser.spaces
                |. Parser.keyword "data"
                |. Parser.symbol ":"
                |. Parser.spaces
                |. Parser.keyword "null"
                |. Parser.spaces
            , Parser.succeed (\rest -> Parser.Done rest)
                |. Parser.keyword "complete"
                |. Parser.spaces
                |. Parser.keyword "data"
                |. Parser.symbol ":"
                |. Parser.spaces
                |= restParser
                |. Parser.spaces
                |. Parser.end
            ]


restParser : Parser.Parser String
restParser =
    Parser.getChompedString <|
        Parser.succeed ()
            |. Parser.chompUntilEndOr "\n"


deadEndsToString : List Parser.DeadEnd -> String
deadEndsToString deadEnds =
    String.concat (List.intersperse "; " (List.map deadEndToString deadEnds))


deadEndToString : Parser.DeadEnd -> String
deadEndToString deadend =
    problemToString deadend.problem ++ " at row " ++ String.fromInt deadend.row ++ ", col " ++ String.fromInt deadend.col


problemToString : Parser.Problem -> String
problemToString p =
    case p of
        Parser.Expecting s ->
            "expecting '" ++ s ++ "'"

        Parser.ExpectingInt ->
            "expecting int"

        Parser.ExpectingHex ->
            "expecting hex"

        Parser.ExpectingOctal ->
            "expecting octal"

        Parser.ExpectingBinary ->
            "expecting binary"

        Parser.ExpectingFloat ->
            "expecting float"

        Parser.ExpectingNumber ->
            "expecting number"

        Parser.ExpectingVariable ->
            "expecting variable"

        Parser.ExpectingSymbol s ->
            "expecting symbol '" ++ s ++ "'"

        Parser.ExpectingKeyword s ->
            "expecting keyword '" ++ s ++ "'"

        Parser.ExpectingEnd ->
            "expecting end"

        Parser.UnexpectedChar ->
            "unexpected char"

        Parser.Problem s ->
            "problem " ++ s

        Parser.BadRepeat ->
            "bad repeat"
